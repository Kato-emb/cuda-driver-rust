/// Defines a wrapper around a sys-level FFI type.
#[macro_export]
macro_rules! wrap_sys_handle {
    ($name:ident, $sys_ty:ty) => {
        #[repr(transparent)]
        #[derive(Clone, Copy)]
        pub struct $name(pub(crate) $sys_ty);

        impl $name {
            pub unsafe fn into_raw(self) -> $sys_ty {
                self.0
            }

            pub unsafe fn from_raw(raw: $sys_ty) -> Self {
                Self(raw)
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self(unsafe { std::mem::zeroed() })
            }
        }

        impl From<$name> for $sys_ty {
            fn from(handle: $name) -> Self {
                handle.0
            }
        }

        impl From<$sys_ty> for $name {
            fn from(raw: $sys_ty) -> Self {
                Self(raw)
            }
        }
    };
}

#[macro_export]
macro_rules! wrap_sys_enum {
    (
        $(#[$enum_attr:meta])*
        $enum_name:ident,
        $sys_type:ty,
        {
            $(
                $(#[$attr:meta])*
                $variant:ident = $sys_variant:ident
            ),* $(,)?
        }
    ) => {
        $(#[$enum_attr])*
        #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
        pub enum $enum_name {
            $(
                $(#[$attr])*
                $variant,
            )*
            __Unknown($sys_type),
        }

        impl From<$sys_type> for $enum_name {
            fn from(value: $sys_type) -> Self {
                match value {
                    $(
                        <$sys_type>::$sys_variant => $enum_name::$variant,
                    )*
                }
            }
        }

        impl From<$enum_name> for $sys_type {
            fn from(value: $enum_name) -> Self {
                match value {
                    $(
                        $enum_name::$variant => <$sys_type>::$sys_variant,
                    )*
                    $enum_name::__Unknown(other) => other,
                }
            }
        }
    };
}
