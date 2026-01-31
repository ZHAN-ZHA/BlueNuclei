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
    name='BlueNuclei',            # produces BlueNuclei.exe
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,                # windowed (no console)
    disable_windowed_traceback=False,

    # Windows icon must be .ico
    icon='app/icon.ico',

    # Optional (leave these as default unless you need them)
    uac_admin=False,
    uac_uiaccess=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BlueNuclei',            # dist/BlueNuclei/BlueNuclei.exe + _internal
)
