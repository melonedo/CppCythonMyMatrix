$configs = 'Debug','RelWithDebInfo','Release'
$build_dir = 'C:/code/build/MyMatrix'
$target = 'tests'
foreach ($c in $configs) {
    Write-Host $c
    # cmake --build $build_dir --config $c -t clean
    cmake --build $build_dir --config $c -j 4 -t MyMatrix-$target
    & "$build_dir/$c/MyMatrix-$target"
}
