project('lda', 'cython', 'c')

py = import('python').find_installation(pure: false)

py.extension_module(
    '_lda',
    files('lda/_lda.pyx', 'lda/gamma.c'),
    dependencies : py.dependency(),
    include_directories : include_directories('lda'),
    install : true,
    install_dir : meson.source_root() / 'lda',
)
