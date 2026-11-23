#include "metal/MetalBackend.hpp"
#include "model/ExpertStore.hpp"
#include "model/GgufPreparation.hpp"
#include "model/Inkling.hpp"
#include "model/InklingImage.hpp"
#include "model/WeightImages.hpp"

#include <bit>
#include <cstring>
#include <exception>
#include <iostream>
#include <iomanip>
#include <string_view>
#include <vector>

static size_t tensorCount(const splash::model::gguf::Image &image) {
  size_t count = 0;
  const auto add = [&](const splash::model::gguf::TensorRows &rows) {
    const auto &source = rows.safetensors;
    count += source.codes ? 1 + bool(source.scales) + bool(source.biases) : 1;
  };
  for (const auto &copy : image.copies) add(copy.source);
  for (const auto &repack : image.repacks)
    for (const auto &source : repack.sources) add(source);
  return count;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: inkling-load METALLIB CHECKPOINT {layer0|all}\n";
    return 2;
  }
  try {
    splash::model::SafetensorsCheckpoint checkpoint(argv[2]);
    const auto image = splash::model::planInklingLayer(checkpoint, 0);
    const std::string_view expert = "language_model.model.layers.0.mlp.switch_mlp.gate_proj";
    const std::string_view query = "language_model.model.layers.0.self_attn.q_proj";
    for (const auto &repack : image.repacks) {
      if (repack.sources.front().name != expert && repack.sources.front().name != query) continue;
      std::vector<uint8_t> row(repack.sources.front().rowBytes);
      splash::model::readRepackRows(repack, 0, 1, row.data());
      uint16_t sb, zb;
      std::memcpy(&sb, row.data(), 2);
      std::memcpy(&zb, row.data() + 2, 2);
      const float scale = std::bit_cast<float>(uint32_t{sb} << 16);
      const float bias = std::bit_cast<float>(uint32_t{zb} << 16);
      std::cout << std::setprecision(9) << (repack.sources.front().name == expert ? "expert gate: " : "query: ")
                << bias + scale * (row[4] & 15)
                << " " << bias + scale * (row[4] >> 4) << '\n';
    }
    splash::metal::MetalBackend backend(argv[1]);
    splash::model::WeightImages images(backend, "inkling-check");
    if (std::string_view(argv[3]) == "layer0") {
      static_cast<void>(splash::model::planInklingLayer(checkpoint, 5));
      static_cast<void>(splash::model::planInklingLayer(checkpoint, 6));
      auto layer = std::make_shared<splash::model::gguf::Image>(image);
      auto embedding = std::make_shared<splash::model::gguf::Image>(splash::model::planInklingEmbedding(checkpoint));
      auto layerFile = splash::model::loadImage(backend, images, "target/", layer, [&] { checkpoint.checkUnchanged(); });
      auto embeddingFile = splash::model::loadImage(backend, images, "target/", embedding,
                                                     [&] { checkpoint.checkUnchanged(); });
      std::cout << "loaded layer 0 and embedding images: " << layerFile.record().declaredBytes << " "
                << embeddingFile.record().declaredBytes << " bytes\n";
      return 0;
    }
    if (std::string_view(argv[3]) != "all") return 2;
    size_t tensors = tensorCount(splash::model::planInklingHead(checkpoint)) +
                     tensorCount(splash::model::planInklingEmbedding(checkpoint));
    for (uint32_t i = 0; i < splash::model::InklingLayout::storedLayers; ++i)
      tensors += tensorCount(splash::model::planInklingLayer(checkpoint, i));
    if (tensors != 1153) throw std::runtime_error("unexpected language tensor count: " + std::to_string(tensors));
    const auto weights = splash::model::loadInklingWeights(backend, {}, images, argv[2], {});
    if (weights.layers.size() != splash::model::InklingLayout::storedLayers || weights.files.size() != 26)
      throw std::runtime_error("unexpected loaded image count");
    std::cout << "loaded " << tensors << " tensors, " << weights.layers.size() << " layers, " << weights.files.size()
              << " images, " << weights.actualAllocatedBytes << " allocated bytes\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
