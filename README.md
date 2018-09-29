## Install
Clone the repository to somewhere, navigate into `my_modules` and enter:

```
pip install -e .
```

This will install the modules into your python environment.

## Create a New Project
Navigate to somewhere you wanna create a new project in and enter:

```
create_project.py -p `your_project_name` -a `your_app_name`
```

For simplicity, assume your project name is `project` and your app name is `app`.

This will create a folder `project` in the directory where you issue the command above. The content of the directory is like:

```
  project
  |__ app
  |   |__ confs
  |   |   |__ __init__.py
  |   |   |__ config_1.py
  |   |
  |   |__ model
  |   |   |__ __init__.py
  |   |   |__ model.py
  |   |
  |   |__ dataset
  |   |   |__ __init__.py
  |   |   |__ dataset.py
  |   |
  |   |__ trainer
  |       |__ __init__.py
  |       |__ trainer.py
  |
  |__ scripts
  |   |__ clean.py
  |   |__ cont.py
  |
  |__ main.py
```

- Write Your Model

  You should create a class named `Model` in `model.py` and subclass `torch.nn.Module`. Refer to `model.py` in the templates to see what's needed in your model code.
  
- Write Your Dataset

  You should create a class named `Dataset` in `dataset.py` and subclass
  `torch.utils.data.Dataset`. Refer to `dataset.py` in the templates to see
  what's needed in your dataset code.
   
- Write Your Trainer

  The trainer is responsible for model checkpoint loading, input data parsing,
  loss computation, logging, checkpoint saving and so on. Most of the core parts
  are implemented within `my_modules.modules.trainer`. Your trainer should
  subclass this core trainer and override its interfaces. Refer to `trainer.py`
  in the templates to see what's needed in your trainer code.
   
- Write a Config File

  The config files are stored in the `confs` directory and named as
  `config_1.py`, `config_2.py`, ... `config_[n].py`, each representing one of
  the configuration (hyper-parameters, data paths, models to use and so on). The
  config is the core part throughout the whole pipeline.
  
  In this manner, you needn't modify the main part of your code in order to apply
  a new configuration. Rather, you just create a new config file and specify it
  when running the program. The config file is in fact a python module defining
  some global variables, and is dynamically loaded during runtime, passed among
  functions like a normal python object.
  
  It's encouraged to create a new config file if a new set of configuration is to
  be tested, while the pipeline (data reading, model processing, etc) remains almost
  the same. However, if the pipeline is changed to some extent but you're still
  doing the same task, it's encouraged to create a separate app in the same proj.
  To do this, simply copy the `app_dir` to your project dir, rename it to your new
  app name, and do the rest just like implementing your first app.
  
  A sample config file is created as you create a new project. You can modify, add
  or remove the variables at your need. But some of them are necessary so that you
  can just modify them.
  
| key              | description                                                                                                                                                                         |
| :--------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PROJECT_ROOT_DIR | The project dir. By default it's the project dir you create. `Leave it unchanged.`                                                                                                  |
| APP_DIR          | The app dir. `Leave it unchanged.`                                                                                                                                                  |
| DATA_DIR         | The directory to place your data (pretrained models, etc). `Leave it unchanged.`                                                                                                    |
| MODEL_DIR        | The directory to store trained model data. For example, model parameters, metric logs and so on. For config_n.py, the data is stored under MODEL_DIR/model_n. `Leave it unchanged.` |
| PRETRAIN_PATH    | The pretrained model path.                                                                                                                                                          |
| STATE_DIR        | The directory to store model parameters. By convention it's a subdirectory of MODEL_DIR. `Leave it unchanged.`                                                                      |
| STATE_PREFIX     | The name prefix of the saved model parameters. They're saved as {STATE_PREFIX}_{epoch_id}.pth. `Leave it unchanged.`                                                                |
| STATE_INDEX      | The `.pth` file to be used in the next train/test phase. Set to `None` if you want to continue from the latest state dict.                                                          |
| SAVE_EPOCH_FREQ  | Frequency to save model weights (in epochs). If `None` or `0`, save every epoch.                                                                                                    |
| MAX_EPOCHS       | Number of epochs to train.                                                                                                                                                          |
| BATCH_SIZE       | A dict specifying batch_size for each mode.                                                                                                                                         |
| PARAM_GROUPS     | See below.                                                                                                                                                                          |
| GPUS             | A list of gpu ids. If more than one gpus are specified, the model will be running on multiple gpus. Otherwise, only use `DEFAULT_GPU`.                                              |
| DEFAULT_GPU      | The default gpu to hold data. When running on multiple gpus, this should be one of the gpu specified in `GPUS`.                                                                     |
| NUM_WORKERS      | Number of workers to use in dataloader.                                                                                                                                             |

*Note: `Leave it unchanged` means that you don't need to modify it since the default setting is enough. You're able to change it but you should take care of where the files are placed.*
  
The `PARAM_GROUPS` specifies how you wanna train you model. It's a list of dictionaries representing different param groups. Each dictionary should has at least two keys: `params` and `lr`. The value of `params` is a list of strings, each representing a submodule of the model. The value of `lr` is the learning rate to be applied to these submodules. If the value of `params` only contains `['default']`, it means all the remaining submodules except for those specified in other param groups. Other keys such as `weight_decay` are available, which depend on the optimizer's settings.
  
Here's an example.

```
PARAM_GROUPS = [{
    'params': ['fc'],
    'lr': 1e-3
  }, {
    'params': ['default'],
    'lr': 1e-4
  }]
```
  
In this example, the `fc` module is trained with a learning rate of `1e-3`, while the rest are trained with `1e-4`. Any module that is not mentioned in the `PARAM_GROUPS` still requires gradient computation, but won't be optimized. Any module whose learning rate is set to 0 neither requires gradient computation nor will be optimized.
  
## Prepare Your Data
By convention you should put your data in the `DATA_DIR` directory, so that the program can find them. Typical data includes:
  
- `pretrained model` When loading model parameters, the trainer proceeds in the following order:
  1. Initialze the model according its `initialize` method.
  2. Look for saved state files (`{STATE_PREFIX}_{epoch_id}.pth`). If found, load it. Otherwise go to 3.
  3. Look for the pretrained model file according to `PRETRAIN_PATH`. If the file exists, load it. Otherwise do nothing.
	   
## Run the Program

```
python main.py -a {app_name} [-c {config_number} [-m {mode}]]
```

Start to train the model of the app `{app_name}` with the `{config_number}` configuration under mode `{mode}`. `{mode}` can be one of `{all, train, test}`. `all` is the default option and means alternatively train and test epochs until `MAX_EPOCHS`. `train` means only train 1 epoch while `test` means only test 1 epoch.
  
## Continue and Clean
```
python scripts/clean -a {app_name} [-c {config_number}]
```
  
Clean the data generated by the `{config_number}` configuration of app `{app_name}`.
  
```
python script/cont {epoch_id} -a {app_name} [-c {config_number}]
```
  
For the `{config_number}` configuration, save the data until the `{epoch_id}` epoch, and delete those after it. Note that, for state files, only the `{epoch_id}` one would be reserved.
