# Android signing continuity

Official distributable APKs for the text Agent branch use the same private signing key from version 2.9 onward.
The key is intentionally not stored in this public repository or on ephemeral CI runners. GitHub Actions builds
an unsigned release APK; the release owner signs it offline and verifies the certificate before distribution.

Expected signing certificate SHA-256:

```text
4B:C8:8E:2E:FE:4D:EC:8E:D3:C8:8D:B9:EC:F6:D7:00:CF:96:D0:F2:46:89:28:71:A7:15:95:57:36:91:18:8F
```

The version 2.8 CI artifact used an ephemeral Android debug key and cannot be upgraded in place to 2.9.
Uninstall it once before installing 2.9. Versions signed with the certificate above can upgrade one another
as long as `versionCode` increases and the application ID remains unchanged.
