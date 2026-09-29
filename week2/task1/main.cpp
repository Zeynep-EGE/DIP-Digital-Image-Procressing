#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>
#include <chrono>

using namespace std::chrono;
using namespace cv;

int main() {
    // Kendi resim yolunu yaz
    std::cout << "OKUNAN YOL: [DEBUG TEST 12345]\n";
    Mat h_img = imread("C:/Users/Kadir/Desktop/DIP-Digital-Image-Procressing/week2/task1/test.jpeg");
    if (h_img.empty()) {
        std::cout << "Resim bulunamadi!\n";
        return 1;
    }

    std::cout << "Orijinal boyut: " << h_img.cols << "x" << h_img.rows << "\n";
    std::cout << "CUDA cihaz sayisi: " << cv::cuda::getCudaEnabledDeviceCount() << "\n\n";

    // ---------- 1) CPU: yeniden boyutlandirma (0.5x ve 2.0x) ----------
    cv::Mat cpu_half, cpu_double, cpu_gray;

    auto t1 = high_resolution_clock::now();
    cv::resize(h_img, cpu_half, cv::Size(), 0.5, 0.5);
    cv::resize(h_img, cpu_double, cv::Size(), 2.0, 2.0);
    cv::cvtColor(h_img, cpu_gray, cv::COLOR_BGR2GRAY);
    auto t2 = high_resolution_clock::now();
    double cpu_ms = duration<double, std::milli>(t2 - t1).count();

    std::cout << "CPU suresi: " << cpu_ms << " ms\n";
    std::cout << "  0.5x boyut: " << cpu_half.cols << "x" << cpu_half.rows << "\n";
    std::cout << "  2.0x boyut: " << cpu_double.cols << "x" << cpu_double.rows << "\n";

    cv::imwrite("cikti_cpu_half.jpg", cpu_half);
    cv::imwrite("cikti_cpu_double.jpg", cpu_double);
    cv::imwrite("cikti_cpu_gray.jpg", cpu_gray);

    // ---------- 2) GPU: ayni islemler CUDA ile ----------
    cv::cuda::GpuMat gpu_img, gpu_half, gpu_double, gpu_gray;

    auto t3 = high_resolution_clock::now();
    gpu_img.upload(h_img);  // veri transferi (PCIe overhead buraya dahil)
    cv::cuda::resize(gpu_img, gpu_half, cv::Size(), 0.5, 0.5);
    cv::cuda::resize(gpu_img, gpu_double, cv::Size(), 2.0, 2.0);
    cv::cuda::cvtColor(gpu_img, gpu_gray, cv::COLOR_BGR2GRAY);

    cv::Mat result_half, result_double, result_gray;
    gpu_half.download(result_half);
    gpu_double.download(result_double);
    gpu_gray.download(result_gray);
    auto t4 = high_resolution_clock::now();
    double gpu_ms = duration<double, std::milli>(t4 - t3).count();

    std::cout << "\nGPU suresi (transfer dahil): " << gpu_ms << " ms\n";

    cv::imwrite("cikti_gpu_half.jpg", result_half);
    cv::imwrite("cikti_gpu_double.jpg", result_double);
    cv::imwrite("cikti_gpu_gray.jpg", result_gray);

    // ---------- 3) Rapor icin ozet ----------
    std::cout << "\n--- KIYASLAMA ---\n";
    std::cout << "CPU: " << cpu_ms << " ms\n";
    std::cout << "GPU: " << gpu_ms << " ms\n";
    std::cout << "Hizlanma: " << (cpu_ms / gpu_ms) << "x\n";

    return 0;
}