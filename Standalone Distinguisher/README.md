### Training and Evaluation of the Standalone Solver

### Data Generator

```shell
python ./g4satbench/generators/generator_simon.py --train_instances 1000 --valid_instances 100 --test_instances 100
```

To change the configuration of the datasets, please  modify the parameter in the code.

### Training

```shell
python train_model.py satisfiability ./simon-12-32-64/anf/train --train_splits sat unsat --valid_dir ./simon-12-32-64/anf/valid --valid_splits sat unsat --label satisfiability --graph anf --model cryptoanfnet --n_iterations 32  --lr 1e-04 --weight_decay 1e-08 --scheduler CosineAnnealingLR --batch_size 8 --seed 123 --epochs 100
```

To change the model, please modify the '--graph' and '--model'. 

### Evaluation

```shell
python eval_model.py satisfiability ./simon-12-32-64/anf/test <path_to_ckpt> --test_splits sat unsat --label satisfiability --graph anf --model cryptoanfnet --n_iterations 32 --batch_size 512
```

To change the model, please modify the '--graph' and '--model'. 