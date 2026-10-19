# config.py

DATASET_CONFIG = {
    'petawawa_cls': {
        'task': 'classification',
        'root': '../data/petawawa/plot_point_clouds',
        'csv': '../data/petawawa/labels.csv',
        'classes': [
            'conif', 
            'decid', 
            'mixed'
        ],
        'label_col': 'dom_sp_type',
        'num_classes': 3
    },
    
    'petawawa_reg': {
        'task': 'regression',
        'root': '../data/petawawa/plot_point_clouds',
        'csv': '../data/petawawa/labels.csv',
        'classes': None,
        'label_col': 'total_agb_z',
        'num_classes': 1
    }
}

MODEL_CONFIG = {
    'pointnet': {
        'model_name': 'pointnet',
        'drop1': 0.4,
        'drop2': 0.0,
        'mat_diff_loss_scale': 0.001,
        'use_normal': True,
        'in_channel': 6,
        
    },
    'pointnet2_msg':{
            'model_name': 'pointnet2_msg',
            'drop1': 0.5,
            'drop2': 0.6,
            'mat_diff_loss_scale': 0.001,
            'use_normal': True,
            'in_channel': 6,
            
    },
    'pointnet2_ssg':{
            'model_name': 'pointnet2_ssg',
            'drop1': 0.4,
            'drop2': 0.4,
            'mat_diff_loss_scale': 0.001,
            'use_normal': True,
            'in_channel': 6,
            
    },
    'pointnext':{
            'model_name': 'pointnext',
            'drop1': 0.5,
            'drop2': 0.6,
            'mat_diff_loss_scale': 0.0, # No regularization for pointnext,
            'use_normal': True,
            'in_channel': 6,
            
    },
    'dgcnn':{
            'model_name': 'dgcnn',
            'drop1': 0.5,
            'drop2': 0.6,
            'k': 20,
            'emb_dims': 512,
            'mat_diff_loss_scale': 0.0, # No regularization for DGCNN,
            'use_normal': False,
            'in_channel': 3,
            
    },
    'ocnn':{
        'model_name': 'ocnn',
        'drop1': 0.1,
        'mat_diff_loss_scale': 0.0, # No regularization for DGCNN,
        'use_normal': False,
        'in_channel': 4,
    }
    
}