package io.vinarytree.liblevenshtein;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Locale;

/** Loads the native library bundled in the JVM artifact. */
final class NativeLibraryLoader {
    private static volatile boolean loaded;

    private NativeLibraryLoader() {}

    static synchronized void load() {
        if (loaded) {
            return;
        }

        String resource = resourcePath(
                System.getProperty("os.name"), System.getProperty("os.arch"));
        try (InputStream input = NativeLibraryLoader.class.getResourceAsStream(resource)) {
            if (input == null) {
                // Source-tree and system-package builds may deliberately provide
                // the library through java.library.path instead of a JAR resource.
                System.loadLibrary("liblevenshtein");
            } else {
                String suffix = resource.substring(resource.lastIndexOf('.'));
                Path extracted = Files.createTempFile("liblevenshtein-", suffix);
                extracted.toFile().deleteOnExit();
                Files.copy(input, extracted, StandardCopyOption.REPLACE_EXISTING);
                System.load(extracted.toAbsolutePath().toString());
            }
            loaded = true;
        } catch (IOException error) {
            throw new ExceptionInInitializerError(error);
        }
    }

    static String resourcePath(String osName, String architecture) {
        String os = osName.toLowerCase(Locale.ROOT);
        String arch = architecture.toLowerCase(Locale.ROOT);
        String normalizedArchitecture = switch (arch) {
            case "amd64", "x86_64", "x64" -> "x86_64";
            case "aarch64", "arm64" -> "aarch64";
            default -> throw unsupported(osName, architecture);
        };

        if (os.contains("linux")) {
            return "/META-INF/native/linux-" + normalizedArchitecture
                    + "/libliblevenshtein.so";
        }
        if ((os.contains("mac") || os.contains("darwin"))
                && normalizedArchitecture.equals("aarch64")) {
            return "/META-INF/native/macos-aarch64/libliblevenshtein.dylib";
        }
        if (os.contains("windows") && normalizedArchitecture.equals("x86_64")) {
            return "/META-INF/native/windows-x86_64/liblevenshtein.dll";
        }
        throw unsupported(osName, architecture);
    }

    private static UnsupportedOperationException unsupported(String os, String arch) {
        return new UnsupportedOperationException(
                "unsupported liblevenshtein JVM platform: " + os + " / " + arch);
    }
}
