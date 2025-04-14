# PSTUN: Perceptive Spectral Transformer Unfolding Network with Multiscale Mixed Training for Arbitrary-scale Hyperspectral and Multispectral Image Fusion

![GitHub](https://img.shields.io/github/license/xwangbin/PSTUN)
![GitHub last commit](https://img.shields.io/github/last-commit/repo-owner/repo-name)

## 📢 NEWS
| Date       | Update                                                                 |
|------------|------------------------------------------------------------------------------------------------------------------------|
| 2025.04.14 | PSTUN has been integrated into the Hyperspectral Image Fusion Toolbox [ HIFTool ](https://github.com/Caoxuheng/HIFtool)          |
| 2025.03.31 | Code released                                                          |
| 2025.03.27 | Paper accepted by *Information Fusion*                                 |

## 📁 Project Structure
| Folder          | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| **PSTUN-main**  | Main module for training/testing different scale HS-MS fusion in simulation experiments |
| **PSTUN-All**   | Proposed multiscale mixed training framework for arbitrary-scale fusion tasks     |
| **PSTUN-Sharpening** | Module for training/testing hyperspectral pansharpening in real-world scenarios |

## 📦 Datasets
| Datasets   |Experiments |  Download                                                                 |
|------------|---------|---------------------------------------------------------------------------|
| Chikusei   |Simulation experiments| [ Download from here ](https://aistudio.baidu.com/datasetdetail/323240/0)       |
| Xiongan    |Simulation experiments| [ Download from here ](https://aistudio.baidu.com/datasetdetail/323240/0)       |
| WorldView3 |Real-world experiments| [ Download from here ](https://github.com/liangjiandeng/PanCollection)          |

## 📝 Usage Notes
1. For technical questions, contact: **wangb@nim.ac.cn**
2. If this repo helps you, please consider citing our works:
   ```bibtex
   @article{PSTUN,
    title={Perceptive Spectral Transformer Unfolding Network with Multiscale Mixed Training for Arbitrary-scale Hyperspectral and Multispectral Image Fusion},
    journal={Information Fusion},
    volume={122},
    year={2025},
    pages={103166},
    issn={1566-2535},
    doi={10.1016/j.inffus.2025.103166},
    url={https://doi.org/10.1016/j.inffus.2025.103166}
}
