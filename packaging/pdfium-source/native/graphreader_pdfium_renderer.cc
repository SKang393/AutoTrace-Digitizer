// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sungwoo Kang

#include <fcntl.h>
#include <io.h>
#include <stdint.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "public/fpdfview.h"

namespace {

constexpr char kMagic[] = {'G', 'R', 'P', 'D', 'F', '0', '1', '\0'};
constexpr int kMaximumDimension = 32768;
constexpr int64_t kMaximumRawBytes = 512LL * 1024LL * 1024LL;

class PdfiumLibrary final {
 public:
  PdfiumLibrary() { FPDF_InitLibrary(); }
  ~PdfiumLibrary() { FPDF_DestroyLibrary(); }
};

class Document final {
 public:
  explicit Document(FPDF_DOCUMENT document) : document_(document) {}
  ~Document() {
    if (document_) {
      FPDF_CloseDocument(document_);
    }
  }
  FPDF_DOCUMENT get() const { return document_; }

 private:
  FPDF_DOCUMENT document_;
};

class Page final {
 public:
  explicit Page(FPDF_PAGE page) : page_(page) {}
  ~Page() {
    if (page_) {
      FPDF_ClosePage(page_);
    }
  }
  FPDF_PAGE get() const { return page_; }

 private:
  FPDF_PAGE page_;
};

class Bitmap final {
 public:
  explicit Bitmap(FPDF_BITMAP bitmap) : bitmap_(bitmap) {}
  ~Bitmap() {
    if (bitmap_) {
      FPDFBitmap_Destroy(bitmap_);
    }
  }
  FPDF_BITMAP get() const { return bitmap_; }

 private:
  FPDF_BITMAP bitmap_;
};

bool ParseInteger(std::wstring_view value,
                  int minimum,
                  int maximum,
                  int* result) {
  if (value.empty()) {
    return false;
  }
  int parsed = 0;
  for (wchar_t character : value) {
    if (character < L'0' || character > L'9') {
      return false;
    }
    const int digit = static_cast<int>(character - L'0');
    if (parsed > (maximum - digit) / 10) {
      return false;
    }
    parsed = parsed * 10 + digit;
  }
  if (parsed < minimum) {
    return false;
  }
  *result = parsed;
  return true;
}

void WriteInt32(std::ofstream& output, int32_t value) {
  const uint8_t bytes[] = {
      static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8),
      static_cast<uint8_t>(value >> 16), static_cast<uint8_t>(value >> 24)};
  output.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

void WriteInt64(std::ofstream& output, int64_t value) {
  std::array<uint8_t, 8> bytes{};
  for (size_t index = 0; index < bytes.size(); ++index) {
    bytes[index] = static_cast<uint8_t>(value >> (index * 8));
  }
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
}

int Fail(const std::string& message, int code) {
  std::cerr << message << std::endl;
  return code;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
  const std::span<wchar_t*> arguments(argv, static_cast<size_t>(argc));
  std::filesystem::path output_path;
  int page_index = -1;
  int dpi = 0;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) {
      return Fail("Every option requires a value.", 2);
    }
    const std::wstring_view option(arguments[index]);
    if (option == L"--output") {
      output_path = arguments[index + 1];
    } else if (option == L"--page") {
      if (!ParseInteger(arguments[index + 1], 0, 1000000, &page_index)) {
        return Fail("Invalid zero-based page index.", 2);
      }
    } else if (option == L"--dpi") {
      if (!ParseInteger(arguments[index + 1], 36, 1200, &dpi)) {
        return Fail("DPI must be between 36 and 1200.", 2);
      }
    } else {
      return Fail("Unknown option.", 2);
    }
  }
  if (output_path.empty() || page_index < 0 || dpi == 0) {
    return Fail("--output, --page, and --dpi are required.", 2);
  }

  if (_setmode(_fileno(stdin), _O_BINARY) == -1) {
    return Fail("Unable to put stdin in binary mode.", 3);
  }
  std::vector<uint8_t> pdf;
  std::array<uint8_t, 64 * 1024> chunk{};
  while (std::cin.good()) {
    std::cin.read(reinterpret_cast<char*>(chunk.data()),
                  static_cast<std::streamsize>(chunk.size()));
    const std::streamsize count = std::cin.gcount();
    if (count > 0) {
      const std::span<const uint8_t> read_chunk(
          chunk.data(), static_cast<size_t>(count));
      pdf.insert(pdf.end(), read_chunk.begin(), read_chunk.end());
    }
  }
  if (pdf.empty()) {
    return Fail("PDF input is empty.", 4);
  }

  PdfiumLibrary library;
  Document document(FPDF_LoadMemDocument64(pdf.data(), pdf.size(), nullptr));
  if (!document.get()) {
    return Fail("PDFium could not load the document. Error=" +
                    std::to_string(FPDF_GetLastError()),
                5);
  }
  const int page_count = FPDF_GetPageCount(document.get());
  if (page_index >= page_count) {
    return Fail("Requested page is outside the document.", 6);
  }
  Page page(FPDF_LoadPage(document.get(), page_index));
  if (!page.get()) {
    return Fail("PDFium could not load the requested page.", 7);
  }

  const double scale = static_cast<double>(dpi) / 72.0;
  const int width = static_cast<int>(std::ceil(FPDF_GetPageWidthF(page.get()) * scale));
  const int height = static_cast<int>(std::ceil(FPDF_GetPageHeightF(page.get()) * scale));
  if (width < 1 || height < 1 || width > kMaximumDimension ||
      height > kMaximumDimension ||
      static_cast<int64_t>(width) * height * 4 > kMaximumRawBytes) {
    return Fail("Rendered page dimensions exceed the safety policy.", 8);
  }

  Bitmap bitmap(FPDFBitmap_Create(width, height, 1));
  if (!bitmap.get()) {
    return Fail("PDFium could not allocate the page bitmap.", 9);
  }
  FPDFBitmap_FillRect(bitmap.get(), 0, 0, width, height, 0xffffffff);
  FPDF_RenderPageBitmap(bitmap.get(), page.get(), 0, 0, width, height, 0,
                        FPDF_ANNOT | FPDF_LCD_TEXT);

  const int stride = FPDFBitmap_GetStride(bitmap.get());
  const int64_t payload_size = static_cast<int64_t>(stride) * height;
  const void* pixels = FPDFBitmap_GetBuffer(bitmap.get());
  if (!pixels || stride < width * 4 || payload_size > kMaximumRawBytes) {
    return Fail("PDFium returned an invalid bitmap.", 10);
  }

  std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
  if (!output) {
    return Fail("Unable to create the raw page output.", 11);
  }
  output.write(kMagic, sizeof(kMagic));
  WriteInt32(output, width);
  WriteInt32(output, height);
  WriteInt32(output, stride);
  WriteInt64(output, payload_size);
  output.write(static_cast<const char*>(pixels), payload_size);
  output.close();
  if (!output) {
    return Fail("Unable to complete the raw page output.", 12);
  }

  std::cout << "OK " << width << " " << height << " " << stride << std::endl;
  return 0;
}
