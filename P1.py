import warnings  # import statements
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")


class GenesAnalyzer:  # Problem 1

        def __init__(self, file_name):
                try:
                        self.patients = pd.read_csv(file_name)  # read csv into pandas df
                except FileNotFoundError:
                        pass

        @staticmethod
        def stats(nums):  # 1A
                n = len(nums)
                mean = sum(nums) / n
                total_diffs = 0.0
                for num in nums:
                        total_diffs = total_diffs + ((num - mean) ** 2.0)  # add squared differences
                sd = (total_diffs / n) ** 0.5  # take square root of variance
                return n, mean, sd

        def gene_type(self, nums, threshold=4.0):  # 1B
                n, mean, sd = self.stats(nums)
                t_score = (mean - threshold) / (sd / (n ** 0.5))
                df = n - 1
                p_value = t.sf(t_score, df)
                bad = p_value < 0.05
                return bad, t_score, p_value

        def analyze_gene(self):  # 1C
                genes = self.patients.iloc[1:, 1:].astype(float)  # take all gene expressions
                analyses = genes.apply(self.gene_type, axis=0)  # vectorized operation of gene_type on each column
                # create dataframe from new dictionary containing analyses data
                analyses_df = pd.DataFrame({'Bad?': analyses.iloc[0, :], 'p-value':
                        analyses.iloc[2, :], 't-score': analyses.iloc[1, :]})
                return analyses_df

        @staticmethod
        def graph_genes(gene_o, gene_t):  # 1D
                plt.scatter(gene_o, gene_t)

                # set ticks incrementing from min to max value of each series
                plt.xticks(ticks=range(int(min(gene_o)) - 1, int(max(gene_o)) + 2))
                plt.yticks(ticks=range(int(min(gene_t)) - 1, int(max(gene_t)) + 2))

                # set title and labels
                plt.title("Gene 1 vs Gene 2 Comparison")
                plt.xlabel('Gene 1')
                plt.ylabel('Gene 2')
                plt.show()


class CellAnalyzer:  # 2B

    def __init__(self, file_path, case_count=10):
        plt.imshow(img.imread('/Users/Arjun/Documents/EDIT ML App/2B Test Cases/0.png'))
        plt.show()
        self.case_count = case_count
        self.threshold_ratio = 0.3
        self.ratios = []
        self.masks = []
        self.process_cells(file_path)

    def process_cells(self, file_path):
        for i in range(self.case_count):
            cell = img.imread(f'{file_path}{i}.png')
            right_cell = cell[:, cell.shape[1] // 2:, :]  # right half of img (mask)
            left_cell = cell[:, :cell.shape[1] // 2, :]  # left half of img (original cell(s))
            self.ratios.append(self.calc_ratios(right_cell))
            self.masks.append(self.segment_cell(left_cell))

    @staticmethod
    def calc_ratios(cell):
        # boolean series checking if pixels match white/green
        w = np.all(cell == (1, 1, 1), axis=2)
        g = np.all(cell == (0, 1, 0), axis=2)

        white = np.sum(w)
        green = np.sum(g)

        blue = cell.shape[0] * cell.shape[1] - white - green  # find blue from pixels not green or white
        return blue / (green + blue)

    def analyze_cell(self):
        _, tscore, pval = GenesAnalyzer('').gene_type(self.ratios, 0.3)
        return "Not Malignant" if pval > 0.05 else "Malignant"

    @staticmethod
    def segment_cell(cell):
        cell_reshape = cell.reshape(-1, 3)  # reshape to contain 3 elements per row
        km = KMeans(n_clusters=3).fit(cell_reshape)  # train kmeans model
        segmented_image = km.cluster_centers_[km.labels_]  # reconfigure pixels to match cluster centers
        segmented_image = segmented_image.reshape(cell.shape)  # convert image back to original shape
        return segmented_image

    def show_cell(self):
        for i in range(self.case_count):
            plt.imshow(self.masks[i])
            plt.show()

file_path = '/Users/Arjun/Documents/EDIT ML App/2B Test Cases/'
analyzer = CellAnalyzer(file_path, case_count=4)
print(analyzer.analyze_cell())
analyzer.show_cell()