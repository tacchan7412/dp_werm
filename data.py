import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import rpy2.robjects.packages
import rpy2.robjects.numpy2ri
import rpy2.robjects.pandas2ri
rpy2.robjects.numpy2ri.activate()
rpy2.robjects.pandas2ri.activate()
edger = rpy2.robjects.packages.importr('edgeR')


def preprocess_data(X, X_test):
    '''
    1. ensure each column to have maximum value of 1
    2. ensure each row to have the (l2-) norm at most 1
        1. calculate (l2-) norm of each row
        2. choose max value
        3. devide all rows by the max value
    '''
    # 1
    X_processed = minmax_scale(X, (-1, 1))
    X_test_processed = minmax_scale(X_test, (-1, 1))
    # 2
    max_norm_x = max(np.linalg.norm(X_processed, axis=1))
    X_processed /= max_norm_x
    max_norm_x_test = max(np.linalg.norm(X_test_processed, axis=1))
    X_test_processed /= max_norm_x_test
    return X_processed, X_test_processed


def generate_synthetic_data(n, m, pos_weight=0.5):
    X, Y = make_classification(2*n, m, weights=[1-pos_weight],
                               random_state=7412)
    Y = (Y * 2 - 1)[:, None]
    X_train, X_test, Y_train, Y_test =\
        train_test_split(X, Y, test_size=n, random_state=7412)
    X_train, X_test = preprocess_data(X_train, X_test)

    return X_train, Y_train, X_test, Y_test


def read_kdd99_df(path):
    df = pd.read_csv(path, header=None, delimiter=",")
    df.columns = ["duration", "protocol_type", "service", "flag", "src_bytes",
                  "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
                  "num_failed_logins", "logged_in", "num_compromised",
                  "root_shell", "su_attempted", "num_root",
                  "num_file_creations", "num_shells", "num_access_files",
                  "num_outbound_cmds", "is_host_login", "is_guest_login",
                  "count", "srv_count", "serror_rate", "srv_serror_rate",
                  "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                  "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
                  "dst_host_srv_count", "dst_host_same_srv_rate",
                  "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                  "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
                  "dst_host_srv_serror_rate", "dst_host_rerror_rate",
                  "dst_host_srv_rerror_rate", "attack"]
    return df


def generate_kdd99_big_data():
    pickle_file = "data/kdd99/kdd99_big.pickle"
    if os.path.exists(pickle_file):
        print('load from', pickle_file)
        with open(pickle_file, 'rb') as f:
            X_train, Y_train, X_test, Y_test = pickle.load(f)
        return X_train, Y_train, X_test, Y_test

    np.random.seed(7412)
    dos_list = ['back.', 'land.', 'neptune.', 'pod.', 'smurf.', 'teardrop.',
                'mailbomb.', 'apache2.', 'processtable.', 'udpstorm.']
    train_df = read_kdd99_df('data/kdd99/raw/kddcup.data_10_percent')
    train_df['is_train'] = 1
    train_df['y'] = 1
    train_df.loc[train_df['attack'].isin(dos_list), 'y'] = -1
    Y_train = train_df['y'].values[:, None]
    train_df.drop('attack', axis=1, inplace=True)
    train_df.drop('y', axis=1, inplace=True)
    test_df = read_kdd99_df('data/kdd99/raw/corrected')
    test_df['is_train'] = 0
    test_df['y'] = 1
    test_df.loc[test_df['attack'].isin(dos_list), 'y'] = -1
    Y_test = test_df['y'].values[:, None]
    test_df.drop('attack', axis=1, inplace=True)
    test_df.drop('y', axis=1, inplace=True)

    df = pd.concat([train_df, test_df], ignore_index=True)
    df = pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'])
    train_df = df[df['is_train'] == 1]
    train_df.drop('is_train', axis=1, inplace=True)
    X_train = train_df.values
    test_df = df[df['is_train'] == 0]
    test_df.drop('is_train', axis=1, inplace=True)
    X_test = test_df.values
    train_ind = np.random.choice(list(range(len(Y_train))),
                                 300000,
                                 replace=False)
    X_train, Y_train = X_train[train_ind], Y_train[train_ind]
    test_ind = np.random.choice(list(range(len(Y_test))),
                                25000,
                                replace=False)
    X_test, Y_test = X_test[test_ind], Y_test[test_ind]

    X_train, X_test = preprocess_data(X_train, X_test)

    with open(pickle_file, 'wb') as f:
        pickle.dump((X_train, Y_train, X_test, Y_test), f)
    return X_train, Y_train, X_test, Y_test


def generate_rnaseq_data():
    '''
    - TCGA
    - GTEx (no disease)
    '''
    pickle_file = "data/rnaseq/rnaseq.pickle"
    if os.path.exists(pickle_file):
        print('load from', pickle_file)
        with open(pickle_file, 'rb') as f:
            X_train, Y_train, X_test, Y_test = pickle.load(f)
        return X_train, Y_train, X_test, Y_test

    geneexpr_TCGA_file = "data/rnaseq/raw/tcga_RSEM_gene_tpm"
    geneexpr_GTEx_file = "data/rnaseq/raw/gtex_RSEM_gene_tpm"
    gene_info_file = "data/TCGA/raw/gencode.v23.annotation.gene.probeMap"

    print('load TCGA gene expression data')
    geneexpr_TCGA = pd.read_csv(geneexpr_TCGA_file,
                                sep='\t',
                                header=0,
                                index_col=0)
    print(" * loaded, size: %d genes, %d samples" % geneexpr_TCGA.shape)

    print('load GTEx gene expression data')
    geneexpr_GTEx = pd.read_csv(geneexpr_GTEx_file,
                                sep='\t',
                                header=0,
                                index_col=0)
    print(" * loaded, size: %d genes, %d samples" % geneexpr_GTEx.shape)

    print('load gencode gene ids to gene names conversion table')
    gene_info = pd.read_csv(gene_info_file, sep='\t', header=0, index_col=0)

    print('find index genes by their names instead of ensembl ids,')
    geneexpr_TCGA.index = pd.Index(gene_info.loc[geneexpr_TCGA.index,
                                   "gene"].values,
                                   name='gene_name')
    geneexpr_GTEx.index = pd.Index(gene_info.loc[geneexpr_GTEx.index,
                                   "gene"].values,
                                   name='gene_name')

    selected_names = ['ARID1A', 'BCORL1', 'CCND1', 'CDH1', 'CHD4', 'CTCF',
                      'ERBB3', 'FBXW7', 'GATA3', 'KMT2C', 'MAP3K1', 'MED12',
                      'NSD1', 'PIK3CA', 'PIK3R1', 'PPP2R1A', 'PTEN', 'RB1',
                      'RNF43', 'RPL22', 'SPOP', 'TP53', 'ZFHX3']

    print('filter out unrelated genes')
    geneexpr_TCGA = geneexpr_TCGA.loc[selected_names]
    print(" * done, size: %d genes left" % geneexpr_TCGA.shape[0])

    print('filter out unrelated genes')
    geneexpr_GTEx = geneexpr_GTEx.loc[selected_names]
    print(" * done, size: %d genes left" % geneexpr_GTEx.shape[0])
    print('merge TCGA and GTEx')
    geneexpr = pd.merge(geneexpr_TCGA, geneexpr_GTEx,
                        how="left", on="gene_name")

    print('invert log transformation')
    tpm = np.maximum(2 ** geneexpr - 0.001, 0)
    del geneexpr

    print('sum the values with the same gene name')
    tpm = tpm.groupby(tpm.index, sort=False).sum()
    print(" * done, %d genes left" % tpm.shape[0])

    print('filter out low expression genes')
    not_low = (np.sum(tpm > 1, axis=1) >= 2)
    tpm = tpm[not_low]
    print(" * filtered, %d genes left" % tpm.shape[0])

    print('filter out samples with zero expression')
    tpm = tpm.loc[:, np.sum(tpm, axis=0) != 0]
    assert not np.any(np.sum(tpm, axis=0) == 0)

    print('calculate normalization factors')
    norm_factors = np.array(edger.calcNormFactors(tpm.values, method="RLE"))

    print('apply normalization factors')
    geneexpr = tpm * norm_factors
    del tpm

    print('redo log-transformation')
    geneexpr = np.log2(geneexpr + 0.001)

    print('transpose so that columns=genes and rows=samples')
    geneexpr = geneexpr.transpose()
    geneexpr.index.name = 'sample_id'

    clinical_data_file = 'data/rnaseq/raw/' +\
        'TCGA_phenotype_denseDataOnlyDownload.tsv'
    print('load clinical data')
    clinical_data = pd.read_csv(clinical_data_file,
                                sep='\t',
                                header=0,
                                index_col=0)
    clinical_data.index.name = 'sample_id'

    print('get only cancer types')
    cancer_type = clinical_data['_primary_disease'].astype('category')
    tmp = pd.merge(geneexpr, cancer_type.to_frame(),
                   how="outer", on='sample_id')
    tmp = tmp.dropna(subset=['ARID1A'])
    tmp['y'] = -1
    disease = 'breast invasive carcinoma'
    tmp.loc[tmp['_primary_disease'] == disease, 'y'] = 1
    disease = 'ovarian serous cystadenocarcinoma'
    tmp.loc[tmp['_primary_disease'] == disease, 'y'] = 1
    disease = 'uterine corpus endometrioid carcinoma'
    tmp.loc[tmp['_primary_disease'] == disease, 'y'] = 1
    disease = 'cervical & endocervical cancer'
    tmp.loc[tmp['_primary_disease'] == disease, 'y'] = 1
    disease = 'uterine carcinosarcoma'
    tmp.loc[tmp['_primary_disease'] == disease, 'y'] = 1

    Y = tmp['y'].values[:, None]
    X = tmp.drop(['_primary_disease', 'y'], axis=1).values

    X_train, X_test, Y_train, Y_test =\
        train_test_split(X, Y, test_size=0.1, random_state=7412)
    X_train, X_test = preprocess_data(X_train, X_test)

    with open(pickle_file, 'wb') as f:
        pickle.dump((X_train, Y_train, X_test, Y_test), f)
    return X_train, Y_train, X_test, Y_test


if __name__ == '__main__':
    X, Y, _, _ = generate_rnaseq_data()
    print(X.shape, sum(Y == 1)/len(Y))
