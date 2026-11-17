# Build of the hotkernels package: the Python sources, and the extension
# module that binds the kernels in ../kernels to NumPy.
# The package's version, and the only place it is written. `uv run release prepare`
# changes it; the release workflow refuses a tag that disagrees with it.
project('hotkernels', 'c', version: '0.1.0',
        default_options: ['c_std=c11', 'buildtype=release', 'warning_level=2'])

py = import('python').find_installation(pure: false)

package = 'hotkernels'
source = 'src' / package

py.install_sources(
  [source / '__init__.py', source / 'environment.py', source / 'hooks.py', source / 'report.py'],
  subdir: package,
)
py.install_sources([source / 'templates' / 'report.txt.j2'], subdir: package / 'templates')

# The kernels are ARM64 assembly for macOS. Anywhere else the package installs
# without the extension, and install() reports that it has nothing to do.
if host_machine.system() == 'darwin' and host_machine.cpu_family() == 'aarch64'
  generate = files('tools/generate.py')
  generated = meson.current_build_dir() / 'generated'
  run_command(py, [generate, 'write', generated], check: true)
  asm_sources = run_command(py, [generate, 'list', 'assembly-sources'], check: true).stdout().strip().split('\n')
  c_sources = run_command(py, [generate, 'list', 'c-sources'], check: true).stdout().strip().split('\n')
  judge_c_flags = run_command(py, [generate, 'list', 'c-flags', '--cpu', get_option('kernel_cpu')], check: true).stdout().strip().split('\n')

  # A kernel that ships as C is its naive reference, compiled with the flags
  # the judge timed it with. The generated header is force-included, as the
  # judge does, so a signature that disagrees with the spec fails the build.
  c_kernels = static_library('c_kernels', c_sources,
    c_args: judge_c_flags + ['-include', generated / 'kernels.h'],
    override_options: ['c_std=c11', 'warning_level=0'],
  )

  numpy_include = run_command(py, ['-c', 'import numpy; print(numpy.get_include())'], check: true).stdout().strip()
  binding = source / '_binding'
  py.extension_module('_binding',
    [binding / 'module.c', binding / 'hook_slot.c', binding / 'hook_loop.c',
     binding / 'hook_function.c', binding / 'hook_method.c', binding / 'hook_index.c'] + asm_sources,
    include_directories: include_directories(numpy_include, is_system: true),
    c_args: ['-O3', '-I' + generated],
    link_with: c_kernels,
    subdir: package,
    install: true,
  )
  install_data(generated / 'kernels.json', install_dir: py.get_install_dir() / package)
endif
