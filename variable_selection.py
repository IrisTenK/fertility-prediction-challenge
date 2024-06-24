# this script select variables
import string
import numpy as np

def select_categorical_var():
    lowercase_alphabet = string.ascii_lowercase
    # this one is categorical:
    religiousRange = ['cr0%s%s012' % (num, letter) if num < 10 else 'cr%s%s012' % (num, letter) for num, letter in
                      zip(np.arange(8, 19), lowercase_alphabet[0:19 - 8])]  # cr08a012 - cr18k012
    # this one is categorical:
    fertilityIntRange = ['cf0%s%s128' % (num, letter) if num < 10 else 'cf%s%s128' % (num, letter) for num, letter in
                         zip(np.arange(8, 21), lowercase_alphabet[0:21 - 8])]  # cf08a128 - cf20m128

    categorical_columns = ['woonvorm_2020', 'oplmet_2020', 'cf20m128', 'cf20m011', 'cs20m330',
                           'burgstat_2020'] + religiousRange + fertilityIntRange

    return(categorical_columns)


def select_features(df):
    lowercase_alphabet = string.ascii_lowercase
    num_list = np.concatenate((np.arange(7, 14), np.arange(15, 21)))
    healthRange = ['ch0%s%s004' % (num, letter) if num < 10 else 'ch%s%s004' % (num, letter) for num, letter in
                   zip(num_list, lowercase_alphabet[0:21 - 8])]  # ch07a004 ch20m004
    # this one is categorical:
    religiousRange = ['cr0%s%s012' % (num, letter) if num < 10 else 'cr%s%s012' % (num, letter) for num, letter in
                      zip(np.arange(8, 19), lowercase_alphabet[0:19 - 8])]  # cr08a012 - cr18k012
    childrenUnderEightRange = ['cw0%s%s439' % (num, letter) if num < 10 else 'cw%s%s439' % (num, letter) for num, letter
                               in zip(np.arange(8, 21), lowercase_alphabet[0:21 - 8])]  # cw08a439- cw20m439
    childrenNumRange = ['cf0%s%s036' % (num, letter) if num < 10 else 'cf%s%s036' % (num, letter) for num, letter in
                        zip(np.arange(8, 15), lowercase_alphabet[0:15 - 8])]  # cf08a036 - cf14g036
    # this one is categorical:
    fertilityIntRange = ['cf0%s%s128' % (num, letter) if num < 10 else 'cf%s%s128' % (num, letter) for num, letter in
                         zip(np.arange(8, 21), lowercase_alphabet[0:21 - 8])]  # cf08a128 - cf20m128
    fertilityNumRange = ['cf0%s%s129' % (num, letter) if num < 10 else 'cf%s%s129' % (num, letter) for num, letter in
                         zip(np.arange(8, 21), lowercase_alphabet[0:21 - 8])]  # cf08a129 - cf20m129
    num_list = np.concatenate((np.arange(8, 14), np.arange(15, 21)))
    mortalityBeliefRange = ['ch0%s%s006' % (num, letter) if num < 10 else 'ch%s%s006' % (num, letter) for num, letter in
                            zip(num_list, lowercase_alphabet[1:21 - 8])]  # ch08b006 - ch20m006

    variable_selection_big =['nomem_encr']+ healthRange + religiousRange + childrenNumRange + childrenUnderEightRange + fertilityIntRange + fertilityNumRange + mortalityBeliefRange
    # +nettoIncomeRange
    # domesticStatusRange+ civilStatusRange+educationRange+stedRange+woonvormRange

    variable_selection_subset = ['cf20m011', 'cf20m003', 'cf20m004', 'cf20m007', 'birthyear_bg', 'cs20m330',
                                 'burgstat_2020', 'oplmet_2020', 'sted_2020', 'woonvorm_2020', 'nettohh_f_2020','cf20m130','cv20l013']

    variable_selection = variable_selection_big + variable_selection_subset
    variable_selection_drop = healthRange + childrenNumRange + childrenUnderEightRange + fertilityNumRange + mortalityBeliefRange

    df = df[variable_selection]

    df['meanHealth'] = df[healthRange].mean(axis=1)
    df['sdHealth'] = df[healthRange].std(axis=1)
    df['meanChildrenUnderEight'] = df[childrenUnderEightRange].mean(axis=1)
    df['sdChildrenUnderEight'] = df[childrenUnderEightRange].std(axis=1)
    df['meanChildrenNum'] = df[childrenNumRange].mean(axis=1)
    df['sdChildrenNum'] = df[childrenNumRange].std(axis=1)
    df['meanChildrenNum'] = df[fertilityNumRange].mean(axis=1)
    df['sdChildrenNum'] = df[fertilityNumRange].std(axis=1)
    df['meanMortalityBelief'] = df[mortalityBeliefRange].mean(axis=1)
    df['sdMortalityBelief'] = df[mortalityBeliefRange].std(axis=1)
    df['mean_fertilityInt'] = df[fertilityIntRange].mean(axis=1)
    df['sd_fertilityInt'] = df[fertilityIntRange].std(axis=1)

    # drop individual years
    df.drop(columns=variable_selection_drop)
    return(df)