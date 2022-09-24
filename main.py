from PeriodicTracer import PeriodicTracer
if __name__ == '__main__':
    boruta_pt = PeriodicTracer(training_data_path='data/malware/training_data',
                                            testing_data_path='data/malware/testing_data',
                                            exp_name='boruta_work')

    boruta_pt.training_build_important_dataset(n_max=4, m_max=1, balance_classes=True, use_omega=False)
    boruta_pt.testing_build_important_dataset(n_max=4, m_max=1)
    boruta_pt.score_boruta_important(n_max=4, m_max=1)