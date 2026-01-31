# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['app/launcher.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('app/templates', 'templates'),
        ('app/static', 'static'),
        ('app/std_scaler.pkl', '.'),
        ('app/minmax_scaler.pkl', '.'),
        ('app/svm_model.pkl', '.'),
        ('app/svm_threshold.pkl', '.'),
        ('app/splash.png', '.'),
        ('version.txt', '.'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tensorflow', 'keras',
        'torch', 'torchvision', 'torchaudio',
        'jax', 'jaxlib',
        'notebook', 'jupyterlab', 'jupyter_server',
        'panel', 'bokeh', 'distributed', 'dask',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BlueNuclei',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='app/icon.icns', 
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BlueNuclei.app', 
)

app = BUNDLE(
    coll,
    name='BlueNuclei.app',
    icon='app/icon.icns',
    bundle_identifier=None
)


