# Yandex Cloud Setup for DVC

## ✅ Step 1 — Install Yandex Cloud CLI

*(If you already have it, skip.)*

**Linux / macOS:**
```bash
curl -sSL https://storage.yandexcloud.net/yandexcloud-yc/install.sh | bash
exec -l $SHELL
```

**Windows:**

Download installer from https://cloud.yandex.com/en/docs/cli/operations/install-cli

---

## ✅ Step 2 — Initialize the CLI

Log into your Yandex Cloud account:

```bash
yc init
```

Choose:
- **Cloud** — your billing cloud
- **Folder** — where you will create the bucket
- **Default region:** `ru-central1`

This creates a working CLI config.

---

## ✅ Step 3 — Create a service account for DVC

This account will own the bucket.

```bash
yc iam service-account create --name dvc-service-account
```

Get its ID:

```bash
yc iam service-account get dvc-service-account
```

---

## ✅ Step 4 — Assign permissions to the service account

The service account must be allowed to work with Object Storage.

```bash
yc resource-manager folder add-access-binding \
  --id $(yc config get folder-id) \
  --role storage.admin \
  --service-account-name dvc-service-account
```

*(You can use `storage.editor` or `storage.uploader`, but for DVC admin is easier.)*

---

## ✅ Step 5 — Create static access keys

These are equivalent to `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`.

```bash
yc iam access-key create --service-account-name dvc-service-account
```

**Output example:**
```yaml
access_key:
  id: aje1...
  key_id: YCAJE...
secret: YCPKl...
```

**Save these:**
- `access_key_id = YCAJE…`
- `secret_key = YCPKl…`

*(We will put them into DVC configuration.)*

---

## ✅ Step 6 — Create an Object Storage bucket

```bash
yc storage bucket create --name agent-042-dvc-data --default-storage-class=standard
```

**Note:**
- Bucket name must be globally unique.
- Region automatically is `ru-central1`.

**To check:**
```bash
yc storage bucket list
```

---

## ✅ Step 7 — Configure DVC to use this bucket

Inside your project:

```bash
dvc remote add -d yc_remote s3://agent-042-dvc-data/dvc
```

Then configure endpoint + region:

```bash
dvc remote modify yc_remote endpointurl https://storage.yandexcloud.net
dvc remote modify yc_remote region ru-central1
```

Add credentials:

```bash
dvc remote modify yc_remote access_key_id YCAJE...
dvc remote modify yc_remote secret_access_key YCPKl...
```

---

## 🟩 Step 8 — Test the integration

```bash
dvc push -v
```

If everything works, you should see uploads happening.

---

## ❤️ You're done

This is the clean production-ready setup.