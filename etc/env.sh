# OpenFOAM v2412 from conda-forge: environment for running the solvers (source after `conda activate candle`)
: "${CONDA_PREFIX:?activate the conda environment first}"
export WM_PROJECT_DIR=$CONDA_PREFIX
export WM_PROJECT_VERSION=v2412
export FOAM_ETC=$CONDA_PREFIX/etc
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/openfoam${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# The conda-forge build resolves libPstream via $ORIGIN/sys-mpich; without it MPI runs fail with
# "The dummy Pstream library cannot be used in parallel mode". Point it at the bundled MPICH Pstream.
if [ ! -e "$CONDA_PREFIX/lib/sys-mpich" ]; then
    mp=$(ls -d "$CONDA_PREFIX"/lib/mpich-* 2>/dev/null | head -1)
    [ -n "$mp" ] && ln -s "$mp" "$CONDA_PREFIX/lib/sys-mpich" && echo "linked $mp -> lib/sys-mpich"
fi
