# Native application dependencies

FDTD Studio's authored C++ and bridge code use this repository's MIT license.
The application dynamically links Qt and VTK; their terms are separate.

* Qt 6.8.3 **qtbase**: Qt Widgets, Core, Gui, OpenGL and Qt Test. The open-source
  distribution contains LGPLv3/GPLv3 and third-party notices. This application
  uses the LGPL-available base libraries, dynamically linked. Qt Charts,
  Qt Data Visualization and other GPL-only Qt add-ons are not required.
  [Qt licensing](https://doc.qt.io/qt-6/licensing.html),
  [Qt 6.8.3 source archive](https://download.qt.io/archive/qt/6.8/6.8.3/single/).
  Keep the supplied licenses, preserve library replacement, and review LGPL
  obligations when redistributing an application bundle.
* VTK 9.5.2: BSD 3-Clause, with separate notices for its bundled third-party
  dependencies. [VTK license](https://vtk.org/about/#license),
  [source archive](https://www.vtk.org/files/release/9.5/VTK-9.5.2.tar.gz).
* aqtinstall (MIT), CMake (BSD), Ninja (Apache-2.0) are build tools, installed
  locally in `.gui-tools`; they are not embedded GUI toolkits.
* Python and the solver's NumPy, SciPy and Shapely dependencies run in a separate
  interpreter. See their installed license notices when bundling that runtime.

The build script copies matching Qt/VTK source license texts, attribution files,
vendor notices and the Qt SDK's SBOM beside the application under
`gui/build/licenses`. The Qt SDK and complete Qt base/VTK source archives
remain in `.gui-deps` on this workstation. The executable currently requires
the repository and a configured Python runtime; this is a developer build,
not a standalone installer bundling Python or the Microsoft C++ runtime.
