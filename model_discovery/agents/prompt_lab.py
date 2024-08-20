import os


import model_discovery.utils as U


dir_path = os.path.dirname(os.path.realpath(__file__))






if __name__ == '__main__':
    from model_discovery.model.gab_composer import GABTree

    ckpt_dir = os.environ['CKPT_DIR']
    db_dir = U.pjoin(ckpt_dir, 'test_composer', 'db')
    test_tree = GABTree('test_tree', db_dir)



