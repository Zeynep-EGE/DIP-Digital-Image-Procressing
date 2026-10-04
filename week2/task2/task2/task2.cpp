#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

int main()
{
  
    const std::string yol = "C:/Users/Kadir/Desktop/DIP-Digital-Image-Procressing/images/test.jpeg";

    cv::Mat original = cv::imread(yol, cv::IMREAD_GRAYSCALE);

    if (original.empty()) {
        std::cerr << "Hata: Resim okunamadi! Aranan yol: " << yol << std::endl;
        std::cin.get();
        return -1;
    }

    std::cout << "Orijinal boyut: " << original.cols << "x" << original.rows << std::endl;

    cv::Mat gray;
    cv::resize(original, gray, cv::Size(1024, 768));

    cv::Mat bit6 = cv::Mat::zeros(gray.size(), CV_8UC1);     
    cv::Mat restored = cv::Mat::zeros(gray.size(), CV_8UC1); 

    for (int i = 0; i < gray.rows; i++) {
        for (int j = 0; j < gray.cols; j++) {
            uchar piksel = gray.at<uchar>(i, j);

            uchar kucuk = piksel / 4;                       
            bit6.at<uchar>(i, j) = kucuk;
            restored.at<uchar>(i, j) = (uchar)(kucuk * 4);  
        }
    }

    cv::imshow("Orijinal (8 bit) 1024x768", gray);
    cv::imshow("6 bit (koyu gorunur)", bit6);
    cv::imshow("6 bit x 4 (basamaklanma)", restored);

    cv::imwrite("gray_1024x768.png", gray);
    cv::imwrite("bit6.png", bit6);
    cv::imwrite("restored.png", restored);

    std::cout << "Dosyalar kaydedildi." << std::endl;

    cv::waitKey(0);
    return 0;
}