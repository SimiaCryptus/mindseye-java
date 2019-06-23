/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.layers.WrapperLayer;

import javax.annotation.Nonnull;
import java.util.List;
import java.util.Map;

@SuppressWarnings("serial")
public class VariableLayer extends WrapperLayer {

  protected VariableLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
  }

  public VariableLayer(final Layer inner) {
    super(inner);
  }

  public static VariableLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new VariableLayer(json, rs);
  }

  @Override
  public List<Layer> getChildren() {
    return super.getChildren();
  }

}
