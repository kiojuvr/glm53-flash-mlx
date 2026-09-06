from setuptools import setup

from mlx import extension


if __name__ == "__main__":
    setup(
        name="glm53-native-execution",
        version="0.0.1",
        description="Probe-only native Metal execution bridge for GLM-5.3",
        ext_modules=[
            extension.CMakeExtension(
                "glm53_native_execution._ext", sourcedir="."
            )
        ],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=["glm53_native_execution"],
        package_data={
            "glm53_native_execution": ["*.so", "*.dylib", "*.metallib"]
        },
        zip_safe=False,
        python_requires=">=3.11",
    )
