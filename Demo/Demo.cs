using System;
using ArrayViewSketches;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace Demo
{
    class Program
    {
        static void Main(string[] args)
        {
            using var context = new Context();
            using var accelerator = new CudaAccelerator(context);
            
            Demo2DDenseX(accelerator);
            Demo2DDenseY(accelerator);
            Demo2DTile(accelerator);
        }

        static void Demo2DDenseX(Accelerator a)
        {
            void Kernel(ArrayView2D<int, Stride2DDenseX> arr, VariableView<int> result)
            {
                // Known at compile time that `v` is dense in the X axis, so the X slice yields a compile time dense 1D view:
                // ArrayView1D<int, DenseDim1D>
                var thirdRow = arr.SliceX(2);
                
                // Not dense in the Y axis, so get a general purpose strided 1D view:
                // ArrayView1D<int, GeneralDim1D>
                var thirdCol = arr.SliceY(2);
                
                // Dummy result to prevent ILGPU pruning the above completely
                result.Value = thirdRow[4] + thirdCol[2];
            }

            // ArrayView2D<int, DenseXDim2D>
            using var buff = a.Allocate2DDenseX<int>((10, 20));
            using var result = a.Allocate<int>(1);
            a.Launch(Kernel, (1, 1), buff.RootView, result.View.GetVariableView(0));
        }
        
        static void Demo2DDenseY(Accelerator a)
        {
            void Kernel<TS>(ArrayView2D<int, TS> arr, VariableView<int> result) where TS : IStride2D
            {
                // Can provide a hint that this is actually dense in the Y dimension if the type doesn't explicitly say so.
                // This should explode if arr.Stride.Y != 1 at runtime.
                var arrY = arr.AsDenseY();
                
                // Not dense in the X axis, so get a general purpose strided 1D view:
                // ArrayView1D<int, GeneralDim1D>
                var thirdRow = arrY.SliceX(2);
                
                
                // Known at compile time that `v` is dense in the Y axis, so the Y slice yields a compile time dense 1D view:
                // ArrayView1D<int, DenseDim1D>
                var thirdCol = arrY.SliceY(2);
                
                // Dummy result to prevent ILGPU pruning the above completely
                result.Value = thirdRow[4] + thirdCol[2];
            }
            
            using var buff = a.Allocate2DDenseY<int>((10, 20));
            using var result = a.Allocate<int>(1);
            a.Launch(Kernel, (1, 1), buff.RootView, result.View.GetVariableView(0));
        }

        static void Demo2DTile(Accelerator a)
        {
            // Defaults to to being Dense in the X dimension
            // ArrayView2D<int, DenseXDim2D>
            using var buff = a.Allocate2D<int>((100, 100));

            // The tile is also dense in the X dimension, and has a Y stride equal the parent view's Y stride
            // ArrayView2D<int, DenseXDim2D>
            var demoTile = buff.RootView.SubView((25, 25), (5, 5));
        }
    }
}