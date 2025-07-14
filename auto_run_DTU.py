import os
import glob


train_base_cmd = 'python train.py -s {} -m {} --eval --iteration 30_000 --nowandb'
render_base_cmd = 'python render.py -s {} -m {} --eval'
metric_base_cmd = 'python metrics.py -m {}'
speed_base_cmd = 'python test_speed.py -s {} -m {}'

size_str = '51.5498176	52.5762328	66.5540608	50.4582592	12.0855696	14.336012	21.0832088	7.987816	25.3373016	18.3500496	22.1314208	10.4557136	23.1431104	12.4014584	11.882444	29.7116112'
scene_str = 'scan24	scan37	scan40	scan55	scan63	scan65	scan69	scan83	scan97	scan105	scan106	scan110	scan114	scan118	scan122'


size_list = [int(float(size) * 0.5) for size in size_str.split('\t')]
scene_list = [s for s in scene_str.split('\t')]
size_dict = {
    s: i for s, i in zip(scene_list, size_list)
}

# Train
dataset_folder = r'/home/liuxf/GS/Datasets/DTU/DTU'
output_folder = 'output/DTU'
folders = glob.glob(os.path.join(dataset_folder, '*'))
for folder in folders:
    if '.' in folder:
        continue

    name = os.path.split(folder)[-1]
    image_cmd = ' -r 2'
    try:
        size_cmd = ' --output_size {}'.format(size_dict[name.lower()])
    except:
        continue

    train_cmd = train_base_cmd.format(
        folder, os.path.join(output_folder, name)
    ) + size_cmd + image_cmd
    print(train_cmd)
    os.system(train_cmd)

    render_cmd = render_base_cmd.format(
        folder, os.path.join(output_folder, name)
    )
    print(render_cmd)
    os.system(render_cmd)

    metric_cmd = metric_base_cmd.format(
        os.path.join(output_folder, name)
    )
    print(metric_cmd)
    os.system(metric_cmd)

    speed_cmd = speed_base_cmd.format(
        folder, os.path.join(output_folder, name)
    )
    print(speed_cmd)
    os.system(speed_cmd)

