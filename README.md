# CKFont2 : Hangul_Font_GenerationModel
## CKFont2: 한글 구성요소를 이용한 개선된 퓨샷 한글 폰트 생성 모델 (2022. 12.)
### CKFont2: An Improved Few-Shot Hangul Font Generation Model based on Hangul Composability

 [KIPS](http://ktsde.kips.or.kr/digital-library/publication?volume=11&number=12)
 KIPS Transactions on Software and Data Engineering, Vol. 11, No. 12, pp. 499-508, Dec. 2022
 [PDF](https://doi.org/10.3745/KTSDE.2022.11.12.499)
---
## Abstract  

A lot of research has been carried out on the Hangeul generation model using deep learning, and recently, research is being carried out how to minimize the number of characters input to generate one set of Hangul (Few-Shot Learning). In this paper, we propose a CKFont2 model using only 14 letters by analyzing and improving the CKFont (hereafter CKFont1) model using 28 letters. The CKFont2 model improves the performance of the CKFont1 model as a model that generates all Hangul using only 14 characters including 24 components (14 consonants and 10 vowels), where the CKFont1 model generates all Hangul by extracting 51 Hangul components from 28 characters. It uses the minimum number of characters for currently known models. From the basic consonants/vowels of Hangul, 27 components such as 5 double consonants, 11/11 compound consonants/vowels respectively are learned by deep learning and generated, and the generated 27 components are combined with 24 basic consonants/vowels. All Hangul characters are automatically generated from the combined 51 components. The superiority of the performance was verified by comparative analysis with results of the zi2zi, CKFont1, and MX-Font model. It is an efficient and effective model that has a simple structure and saves time and resources, and can be extended to Chinese, Thai, and Japanese.  

---  
## Method

### - CKFont2 concept diagram
<img src = "https://user-images.githubusercontent.com/62954678/184546629-9831c690-3f11-456f-821f-0988e7b40f7f.png" width="800" height = "300"> 

### - Sample output images of the double and compound characters left and characters right generated  

   <img src = "https://user-images.githubusercontent.com/62954678/184546662-2f2f4bd9-0262-48e9-85cf-778d1bdcf6a6.png" width="500" height = "300">       <img src = "https://user-images.githubusercontent.com/62954678/184546679-3e39952e-17db-4421-8aa6-5cfe0d9d83be.png" width="500" height = "300"> 
---
## Results  

### - Sample output images with other models  

<img src = "https://user-images.githubusercontent.com/62954678/184548161-f354b83e-43a6-42ea-a0b5-f9bd330431e0.png" width="1000" height = "400"> 

## Citation
---
J. Park, A. U. Hassan and J. Choi, "An Improved Few-Shot Hangul Font Generation Model based on Hangul Composability", KIPS Transactions on Software and Data Engineering, vol. . DOI: https://doi.org/.  
---
## Copyright
---
The code and other helping modules are only allowed for PERSONAL and ACADEMIC usage.
---
