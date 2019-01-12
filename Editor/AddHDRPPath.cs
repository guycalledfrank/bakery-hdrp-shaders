#if UNITY_EDITOR

using System.Linq;
using UnityEngine;
using System.IO;

namespace UnityEditor.Experimental.Rendering
{
    static class AddHDRPPath
    {
#if UNITY_2018_1_OR_NEWER
        [ShaderIncludePath]
#endif
        public static string[] GetPaths()
        {
            return new string [] {Path.GetFullPath("Packages/com.unity.render-pipelines.high-definition")};
        }
    }
}

#endif
