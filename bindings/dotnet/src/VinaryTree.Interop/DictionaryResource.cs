using System.Runtime.InteropServices;

namespace VinaryTree.Interop;

/// <summary>The stable two-machine-word retained resource descriptor.</summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct NativeResource
{
    internal readonly nint Context;
    internal readonly nint VTable;

    /// <summary>True when either required ABI word is null.</summary>
    public bool IsNull => Context == 0 || VTable == 0;
}

/// <summary>A callback borrowing a producer-owned resource for one native call.</summary>
public unsafe delegate TResult ResourceCallback<TResult>(NativeResource* resource);

/// <summary>Implemented by modular packages that provide retained native resources.</summary>
public interface IDictionaryResource
{
    /// <summary>
    /// Keep the producer handle alive and lend its two-word descriptor to a
    /// synchronous callback. The callback must not retain the descriptor
    /// itself; a native consumer may retain it through its vtable.
    /// </summary>
    unsafe TResult WithResource<TResult>(ResourceCallback<TResult> callback);
}
