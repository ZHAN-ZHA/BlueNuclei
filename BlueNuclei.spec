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
        ('version.txt', '.'),
        ('app/splash.png', '.'),  
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BlueNuclei',
)

app = BUNDLE(coll,
         name='BlueNuclei.app',
         icon=None,
         bundle_identifier=None)