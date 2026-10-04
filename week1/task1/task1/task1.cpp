#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Diskten resmi oku
    cv::Mat image = cv::imread("C:/Users/Kadir/Desktop/DIP-Digital-Image-Procressing/images/test.jpeg");

    if (image.empty()) {
        std::cerr << "Hata: Resim okunamadi! Yolu kontrol et." << std::endl;
        return -1;
    }

    std::cout << "Orijinal boyut: " << image.cols << "x" << image.rows << std::endl;

    // 1024x768 boyutuna getir
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(1024, 768));

    // Kaydet
    cv::imwrite("resim_1024x768.jpg", resized);

    // Göster
    cv::imshow("1024x768 Goruntu", resized);
    cv::waitKey(0);

    return 0;
}