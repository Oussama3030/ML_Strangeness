"""Run the whole bias investigation.  Nothing outside Claude/ is written."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MACROS = ['bias_01_domain_shift.py',
          'bias_02_feature_audit.py',
          'bias_03_kinematics.py',
          'bias_04_mass_sculpting.py',
          'bias_05_pipeline_audit.py',
          'bias_06_honest_number.py',
          'bias_07_pt_dependence.py',
          'bias_08_ptcut_vs_reweight.py',
          'bias_10_profiles.py']

fail = 0
for m in MACROS:
    print('\n' + '#' * 78)
    print('# ' + m)
    print('#' * 78)
    r = subprocess.run([sys.executable, os.path.join(HERE, m)], cwd=HERE)
    if r.returncode != 0:
        fail += 1
        print('[FAILED] ' + m)

print('\nAll output in {}/output/'.format(HERE))
sys.exit(1 if fail else 0)
