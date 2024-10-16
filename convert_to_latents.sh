: '
Script for downloading, cleaning and resizing Open X-Embodiment Dataset (https://robotics-transformer-x.github.io/)

Performs the preprocessing steps:
  1. Downloads mixture of 25 Open X-Embodiment datasets
  2. Runs resize function to convert all datasets to 256x256 (if image resolution is larger) and jpeg encoding
  3. Fixes channel flip errors in a few datsets, filters success-only for QT-Opt ("kuka") data

To reduce disk memory usage during conversion, we download the datasets 1-by-1, convert them
and then delete the original.
We specify the number of parallel workers below -- the more parallel workers, the faster data conversion will run.
Adjust workers to fit the available memory of your machine, the more workers + episodes in memory, the faster.
The default values are tested with a server with ~120GB of RAM and 24 cores.
'

SOURCE_DIR='/mnt/data/erbauer/oxe_dataset/downloads/'
TARGET_DIR='/mnt/data1/erbauer/oxe_latents/converted'
N_WORKERS=20               # number of workers used for parallel conversion --> adjust based on available RAM
MAX_EPISODES_IN_MEMORY=100 # number of episodes converted in parallel --> adjust based on available RAM

# increase limit on number of files opened in parallel to 20k --> conversion opens up to 1k temporary files
# in /tmp to store dataset during conversion
ulimit -n 20000

# format: [dataset_name, dataset_version, transforms]
DATASET_TRANSFORMS=(
	# "fractal20220817_data 0.1.0 resize_and_jpeg_encode"
	# "bridge 0.1.0 resize_and_jpeg_encode"
	# "kuka 0.1.0 resize_and_jpeg_encode,filter_success"
	# "taco_play 0.1.0 resize_and_jpeg_encode"
	# "jaco_play 0.1.0 resize_and_jpeg_encode"
	# "berkeley_cable_routing 0.1.0 resize_and_jpeg_encode"
	# "roboturk 0.1.0 resize_and_jpeg_encode"
	# "nyu_door_opening_surprising_effectiveness 0.1.0 resize_and_jpeg_encode"
	# "viola 0.1.0 resize_and_jpeg_encode"
	# "berkeley_autolab_ur5 0.1.0 resize_and_jpeg_encode,flip_wrist_image_channels"
	# "toto 0.1.0 resize_and_jpeg_encode"
	# "language_table 0.1.0 resize_and_jpeg_encode"
	# "stanford_hydra_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode,flip_wrist_image_channels,flip_image_channels"
	# "austin_buds_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode"
	# "nyu_franka_play_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode"
	# "furniture_bench_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode"
	# "ucsd_kitchen_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode"
	# "austin_sailor_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode"
	# "austin_sirius_dataset_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode"
	# "bc_z 0.1.0 resize_and_jpeg_encode"
	# "dlr_edan_shared_control_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode"
	# "iamlab_cmu_pickup_insert_converted_externally_to_rlds 0.1.0 resize_and_jpeg_encode"
	# "utaustin_mutex 0.1.0 resize_and_jpeg_encode,flip_wrist_image_channels,flip_image_channels"
	# "berkeley_fanuc_manipulation 0.1.0 resize_and_jpeg_encode,flip_wrist_image_channels,flip_image_channels"
	# "cmu_stretch 0.1.0 resize_and_jpeg_encode"
	"faive_bottle_pick_dataset 1.0.0 encode_gc_angles_to_latent"
	"faive_cube_sort_dataset 1.0.0 encode_gc_angles_to_latent"
	"faive_pokemon_pick_dataset 1.0.0 encode_gc_angles_to_latent"
	"faive_towel_wipe_dataset 1.0.0 encode_gc_angles_to_latent"
	"faive_plush_pick_dataset 1.0.0 encode_gc_angles_to_latent"
	"dexycb_dataset 1.0.0 encode_mano_params_to_latent"
	"bridge_dataset 1.0.0 encode_simple_gripper_to_latent"
)

for tuple in "${DATASET_TRANSFORMS[@]}"; do
	# Extract strings from the tuple
	strings=($tuple)
	DATASET=${strings[0]}
	VERSION=${strings[1]}
	TRANSFORM=${strings[2]}
	python3 modify_rlds_dataset.py --dataset=$DATASET --data_dir=$SOURCE_DIR --target_dir=$TARGET_DIR --mods=$TRANSFORM --n_workers=$N_WORKERS --max_episodes_in_memory=$MAX_EPISODES_IN_MEMORY
done
