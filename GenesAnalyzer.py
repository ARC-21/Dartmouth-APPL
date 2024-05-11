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

    @staticmethod
    def gene_type(nums, threshold=4.0):  # 1B
        n, mean, sd = stats(nums)
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
# Problem 1 Output Code
file_name = '/Users/Arjun/Documents/EDIT ML App/1_c_d.csv'
analyzer = GenesAnalyzer(file_name)

# get gene expressions
gene_one = analyzer.patients.loc[:, 'sample_0']
gene_two = analyzer.patients.loc[:, 'sample_1']

analyzer.graph_genes(gene_one, gene_two)
analyzer.analyze_gene()


class DiseaseSubtype:  # 2A

    def __init__(self, file_name):
        self.y, self.X = self.load_data(file_name)

    @staticmethod
    def load_data(file_name):
        patients = pd.read_csv(file_name)
        y = patients.to_numpy()[:, 0].astype(str)  # set y to an ndarray of target variables (ALL or AML)
        y = np.where(np.char.find(y, 'ALL') != -1, 0, 1).astype(int)  # set variables to 0 or 1
        X = StandardScaler().fit_transform(patients.iloc[:, 1:])  # scale to reduce bias
        return y, X

    def plot(self):
        pca = PCA(n_components=2)
        X_PCA = pca.fit_transform(self.X)
        subtype_assignments = np.where(self.y == 0, 'ALL (Lymphoid)', 'AML (Myeloid)')  # ALL if 0 else 1
        plt.figure(figsize=(8, 6))

        for subtype in np.unique(subtype_assignments):
            self.scatter(X_PCA, subtype_assignments, subtype)

        plt.title('Leukemia Patient Clustering Based on Gene Expression Profiles')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend(title='Disease Subtypes')
        plt.show()

    @staticmethod
    def scatter(X, subtype_assignments, subtype):
        # filter our non-matching lineage from X
        x_coords_all = X[subtype_assignments == subtype, 0]
        y_coords_all = X[subtype_assignments == subtype, 1]
        plt.scatter(x_coords_all, y_coords_all, label=subtype)

    def assess_subtype(self):
        knn = KNeighborsClassifier(n_neighbors=9)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.15, random_state=42)
        knn.fit(X_train, y_train)
        return 100 * knn.score(X_test, y_test)
file_name = '/Users/Arjun/Documents/EDIT ML App/2_a.csv'
analyzer = DiseaseSubtype(file_name)
analyzer.plot()
print('Subtype Assessment Accuracy: ' + str(analyzer.assess_subtype()) + '%')


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
'''
First Cell:
    - Downloads libraries
Second Cell:
    - Fetches all of 20 Newsgroups dataset, while ensuring samples are randomized and randomization is reproducible
Third Cell:
    - Returns number of documents in the dataset in an f-string
    - Returns number of categories in the document in an f-string
Fourth Cell:
    - Retrieves attribute containing names of the categories/topics in the dataset
Fifth Cell:
    - Iterates over first three documents with each element being in the form of index, document
    - Retrieves category of current document
    - Prints first 500 characters of each document while displaying its category
Sixth Cell:
    - Creates a list of category names for each target label
    - Creates a pandas DataFrame containing a single column, comprised of the category names
Seventh Cell:
    - Creates a matrix of token counts, each row representing a document and each column representing a unique word, with English stopwords removed and the minimum word count being 5
    - Creates a matrix of the counts of each word in the document vocabulary, each row representing a document and each column representing a word count
Eighth Cell:
    - Displays contents of 'word_doc_matrix'
Ninth Cell:
    - Dimensionality of 'word_doc_matrix' reduced to 2 dimensions for visualization purposes using Hellinger distance
Tenth Cell:
    - Displays shape of embedded dimension-reduced data
Eleventh Cell:
    - Creates a scatterplot of embedded datapoints, with labels displayed alongside data points
Twelfth Cell:
    - Copy of Seventh Cell, but utilizes TfidfVectorizer instead of CountVectorizer to create a matrix using td-idf weighting
Thirteenth Cell:
    - Displays content of 'tdif_word_doc_matrix'
Fourteenth Cell:
    - Copy of Ninth Cell, but uses tdidf_word_doc_matrix
Fifteenth Cell:
    - Copy of Eleventh Cell, but uses tdidf_word_doc_matrix
'''