cd src/custom_nodes
# make NODES=image_preprocessing
make
cp lib/ubuntu22/libcustom_node_*.so ../../models/
cd -
