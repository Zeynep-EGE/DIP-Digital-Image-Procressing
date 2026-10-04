#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

int main()
{
    const std::string yol = "C:/Users/Kadir/Desktop/DIP-Digital-Image-Procressing/images/test.jpeg";

    cv::Mat gray = cv::imread(yol, cv::IMREAD_GRAYSCALE);

    if (gray.empty()) {
        std::cerr << "Hata: Resim okunamadi! Aranan yol: " << std::endl;
        std::cin.get();
        return -1;
    }

    std::cout << "Orijinal boyut: " << gray.cols << "x" << gray.rows << std::endl;

    int seviyeler[] = { 64, 16, 8, 4, 2 };

    cv::namedWindow("Orjinal gri (256 ton)", cv::WINDOW_NORMAL);
    cv::imshow("Orijinal gri (256 ton)", gray);

    for (int L : seviyeler) {
        cv::Mat quant = cv::Mat::zeros(gray.size(), CV_8UC1);
        int aralik = 256 / L;

        for (int i = 0; i < gray.rows; i++) {
            for (int j = 0; j < gray.cols; j++) {
                int piksel = gray.at<uchar>(i, j);

                int basamak = piksel / aralik;              // hangi tona düştü
                int yeni = basamak * 255 / (L - 1);         // 0-255 aralığına geri yay

                quant.at<uchar>(i, j) = (uchar)yeni;
            }
        }

        std::string ad = std::to_string(L) + " ton";
        cv::namedWindow(ad, cv::WINDOW_NORMAL);
        cv::imshow(ad, quant);
        cv::imwrite("quant_" + std::to_string(L) + ".png", quant);

        std::cout << L << " seviyeli kuantizasyon kaydedildi." << std::endl;
    }

    cv::waitKey(0);
    return 0;
}



