using System;

namespace ArrayViewSketches
{
    public readonly struct DimTuple1
    {
        public DimTuple1(long x)
        {
            X = x;
        }

        public long X { get; }

        public static implicit operator DimTuple1(long v) => new DimTuple1(v);
        public static implicit operator DimTuple1(ValueTuple<long> v) => new DimTuple1(v.Item1);
    }
    
    public readonly struct DimTuple2
    {
        public DimTuple2(long x, long y)
        {
            X = x;
            Y = y;
        }

        public long X { get; }
        public long Y { get; }

        public static implicit operator DimTuple2(ValueTuple<long, long> v) => new DimTuple2(v.Item1, v.Item2);
    }
}