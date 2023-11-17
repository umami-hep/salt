INSTDIR=python_install
rm -rf ${INSTDIR}
mkdir ${INSTDIR}
export PYTHONPATH=${PWD}:${PWD}/${INSTDIR}:${PYTHONPATH}
rm -rf salt.egg-*
python -m pip install --prefix ${INSTDIR} -e .
export PATH=${PWD}/${INSTDIR}/bin:$PATH
